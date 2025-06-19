import { useRoute } from "wouter";
import { ArrowLeft, ExternalLink, Settings, Code, AlertCircle, BookOpen } from "lucide-react";
import { Link } from "wouter";
import Header from "@/components/Header";
import Footer from "@/components/Footer";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Separator } from "@/components/ui/separator";
import { applications } from "@/data/applications";

export default function ApplicationDetail() {
  const [, params] = useRoute("/application/:slug");
  const slug = params?.slug;
  
  const application = applications.find(app => app.slug === slug);
  
  if (!application) {
    return (
      <div className="min-h-screen bg-white">
        <Header />
        <div className="max-w-7xl mx-auto px-4 py-12">
          <Card className="max-w-md mx-auto">
            <CardContent className="pt-6 text-center">
              <AlertCircle className="h-12 w-12 text-red-500 mx-auto mb-4" />
              <h1 className="text-2xl font-bold text-gray-900 mb-2">Application Not Found</h1>
              <p className="text-gray-600 mb-6">The requested AI application could not be found.</p>
              <Button asChild>
                <Link href="/">Return to Home</Link>
              </Button>
            </CardContent>
          </Card>
        </div>
        <Footer />
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-white">
      <Header />
      
      <div className="max-w-7xl mx-auto px-4 py-8">
        {/* Breadcrumb */}
        <div className="flex items-center mb-8">
          <Button variant="ghost" asChild className="mr-4">
            <Link href="/">
              <ArrowLeft className="h-4 w-4 mr-2" />
              Back to Matrix
            </Link>
          </Button>
          <div className="text-sm text-gray-500">
            AI Applications / {application.area}
          </div>
        </div>

        {/* Header */}
        <div className="mb-8">
          <Badge variant="secondary" className="mb-2">{application.category}</Badge>
          <h1 className="text-4xl font-bold text-gov-navy mb-4">{application.area}</h1>
          <p className="text-xl text-gray-600 leading-relaxed">{application.description}</p>
        </div>

        <div className="grid lg:grid-cols-3 gap-8">
          {/* Main Content */}
          <div className="lg:col-span-2 space-y-8">
            {/* Overview */}
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center">
                  <BookOpen className="h-5 w-5 mr-2" />
                  Overview
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div>
                  <h3 className="font-semibold text-gov-navy mb-2">Practical Examples</h3>
                  <p className="text-gray-700">{application.example}</p>
                </div>
                <Separator />
                <div>
                  <h3 className="font-semibold text-gov-navy mb-2">Technical Description</h3>
                  <p className="text-gray-700">{application.detailedDescription}</p>
                </div>
              </CardContent>
            </Card>

            {/* Implementation Guide */}
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center">
                  <Settings className="h-5 w-5 mr-2" />
                  Implementation Guide
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-6">
                <div>
                  <h3 className="font-semibold text-gov-navy mb-3">Required Hardware</h3>
                  <ul className="list-disc list-inside space-y-1 text-gray-700">
                    {application.hardware.map((item, index) => (
                      <li key={index}>{item}</li>
                    ))}
                  </ul>
                </div>
                
                <div>
                  <h3 className="font-semibold text-gov-navy mb-3">Required Software</h3>
                  <ul className="list-disc list-inside space-y-1 text-gray-700">
                    {application.software.map((item, index) => (
                      <li key={index}>{item}</li>
                    ))}
                  </ul>
                </div>

                <div>
                  <h3 className="font-semibold text-gov-navy mb-3">Step-by-Step Setup</h3>
                  <ol className="list-decimal list-inside space-y-2 text-gray-700">
                    {application.setup.map((step, index) => (
                      <li key={index} className="leading-relaxed">{step}</li>
                    ))}
                  </ol>
                </div>
              </CardContent>
            </Card>

            {/* Code Examples */}
            {application.codeExample && (
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center">
                    <Code className="h-5 w-5 mr-2" />
                    Code Example
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <pre className="bg-gray-100 p-4 rounded-lg overflow-x-auto text-sm">
                    <code>{application.codeExample}</code>
                  </pre>
                </CardContent>
              </Card>
            )}

            {/* Troubleshooting */}
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center">
                  <AlertCircle className="h-5 w-5 mr-2" />
                  Troubleshooting
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  {application.troubleshooting.map((item, index) => (
                    <div key={index}>
                      <h4 className="font-medium text-gov-navy mb-1">{item.issue}</h4>
                      <p className="text-gray-700 text-sm">{item.solution}</p>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          </div>

          {/* Sidebar */}
          <div className="space-y-6">
            {/* Quick Info */}
            <Card>
              <CardHeader>
                <CardTitle>Quick Information</CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div>
                  <h4 className="font-semibold text-gov-navy mb-1">Difficulty Level</h4>
                  <Badge variant={application.difficulty === 'Beginner' ? 'secondary' : 
                                 application.difficulty === 'Intermediate' ? 'default' : 'destructive'}>
                    {application.difficulty}
                  </Badge>
                </div>
                <div>
                  <h4 className="font-semibold text-gov-navy mb-1">Implementation Time</h4>
                  <p className="text-gray-700">{application.implementationTime}</p>
                </div>
                <div>
                  <h4 className="font-semibold text-gov-navy mb-1">Cost Estimate</h4>
                  <p className="text-gray-700">{application.cost}</p>
                </div>
              </CardContent>
            </Card>

            {/* Sources */}
            <Card>
              <CardHeader>
                <CardTitle>Sources & References</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-2">
                  {application.sourceLinks.map((source, index) => (
                    <a
                      key={index}
                      href={source.url}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="flex items-center text-gov-blue hover:underline text-sm"
                    >
                      <ExternalLink className="h-3 w-3 mr-2" />
                      {source.title}
                    </a>
                  ))}
                </div>
              </CardContent>
            </Card>

            {/* Related Applications */}
            <Card>
              <CardHeader>
                <CardTitle>Related Applications</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-2">
                  {applications
                    .filter(app => app.category === application.category && app.slug !== application.slug)
                    .slice(0, 3)
                    .map((relatedApp) => (
                      <Link key={relatedApp.slug} href={`/application/${relatedApp.slug}`}>
                        <div className="text-sm text-gov-blue hover:underline cursor-pointer">
                          {relatedApp.area}
                        </div>
                      </Link>
                    ))}
                </div>
              </CardContent>
            </Card>
          </div>
        </div>
      </div>
      
      <Footer />
    </div>
  );
}
