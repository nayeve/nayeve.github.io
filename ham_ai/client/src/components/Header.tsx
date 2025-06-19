import { useState } from "react";
import { Link } from "wouter";
import { Button } from "@/components/ui/button";
import { Menu, X, Shield, Star } from "lucide-react";

export default function Header() {
  const [isMenuOpen, setIsMenuOpen] = useState(false);

  return (
    <header className="bg-slate-800 text-white">
      {/* Top Government Bar */}
      <div className="bg-slate-700 py-2 text-sm">
        <div className="max-w-7xl mx-auto px-4 flex justify-between items-center">
          <div className="flex items-center space-x-4">
            <span className="flex items-center">
              <Shield className="h-4 w-4 mr-2" />
              An official website of the Jay States Communications Authority
            </span>
          </div>
          <div className="flex items-center space-x-4">
            <span className="flex items-center text-green-400">
              <div className="w-2 h-2 bg-green-400 rounded-full mr-2"></div>
              Secure .us
            </span>
          </div>
        </div>
      </div>

      {/* Main Header */}
      <div className="max-w-7xl mx-auto px-4 py-4">
        <div className="flex items-center justify-between">
          <Link href="/">
            <div className="flex items-center space-x-4 cursor-pointer">
              {/* Government Seal */}
              <div className="w-16 h-16 bg-white rounded-full flex items-center justify-center">
                <Star className="text-slate-800 h-8 w-8" />
              </div>
              <div>
                <h1 className="text-xl font-bold">Jay States Communications Authority</h1>
                <p className="text-sm opacity-90">Department of Communications Technology</p>
              </div>
            </div>
          </Link>
          
          <nav className="hidden md:flex space-x-6">
            <Link href="/" className="hover:text-blue-300 transition-colors">Home</Link>
            <a href="#applications" className="hover:text-blue-300 transition-colors">Applications</a>
            <a href="#sources" className="hover:text-blue-300 transition-colors">Sources</a>
            <a href="#author" className="hover:text-blue-300 transition-colors">Author</a>
          </nav>
          
          <Button
            variant="ghost"
            size="sm"
            className="md:hidden text-white hover:bg-slate-700"
            onClick={() => setIsMenuOpen(!isMenuOpen)}
          >
            {isMenuOpen ? <X className="h-5 w-5" /> : <Menu className="h-5 w-5" />}
          </Button>
        </div>
        
        {/* Mobile Menu */}
        {isMenuOpen && (
          <nav className="md:hidden mt-4 pb-4">
            <div className="flex flex-col space-y-3">
              <Link href="/" className="hover:text-blue-300 transition-colors">Home</Link>
              <a href="#applications" className="hover:text-blue-300 transition-colors">Applications</a>
              <a href="#sources" className="hover:text-blue-300 transition-colors">Sources</a>
              <a href="#author" className="hover:text-blue-300 transition-colors">Author</a>
            </div>
          </nav>
        )}
      </div>
    </header>
  );
}
